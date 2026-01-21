import React from 'react';

interface BotIconProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export const BotIcon: React.FC<BotIconProps> = ({ className = '', size = 'md' }) => {
  const sizeMap = {
    sm: 'w-5 h-5',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-10 h-10'
  };

  return (
    <img
      src="/bot-icon.png?v=1768668000"
      alt="AI Bot"
      className={`${sizeMap[size]} ${className} object-contain`}
    />
  );
};
